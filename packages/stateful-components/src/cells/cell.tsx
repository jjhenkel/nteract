import React from "react";
import { connect } from "react-redux";

import { ImmutableCell } from "@nteract/commutable/src";
import { AppState, selectors } from "@nteract/core";

interface ComponentProps {
  id: string;
  contentRef: string;
  children: React.ReactNode;
}

interface StateProps {
  cell?: ImmutableCell;
  selected: boolean;
  status: string;
}

type Props = ComponentProps & StateProps;

export class Cell extends React.Component<Props> {
  shouldComponentUpdate(nextProps: Props): boolean {
    // Update cell directly after converting between markdown and code
    if (this.props.cell?.cell_type !== nextProps.cell?.cell_type) {

      return true;
    }
    // Update cell when selecting or de-selecting it
    if (nextProps?.selected !== this.props?.selected) {

      return true
    }
    // Update cell when changing status
    if (nextProps?.status !== this.props?.status) {

      return true
    }
    return false
  }

  render(): JSX.Element | null {
    // We must pick only one child to render
    let chosenOne: React.ReactNode | null = null;

    if (!this.props.cell) {
      return null;
    }

    const cell_type = this.props.cell.get("cell_type", "code");

    // Find the first child element that matches something in this.props.data
    React.Children.forEach(this.props.children, child => {
      if (!child) {
        return;
      }

      if (
        typeof child === "string" ||
        typeof child === "number" ||
        typeof child === "boolean"
      ) {
        return;
      }

      if (chosenOne) {
        // Already have a selection
        return;
      }

      if (typeof child !== "object" || !("props" in child)) {
        return;
      }

      if (child.props && child.props.cell_type) {
        const child_cell_type = child.props.cell_type;

        chosenOne = child_cell_type === cell_type ? child : null;
        return;
      }
    });

    // If we didn't find a match, render nothing
    if (chosenOne === null) {
      return null;
    }

    // Render the output component that handles this output type
    return (
      <div
        className={`nteract-cell-container ${
          this.props.selected ? "selected" : ""
        }`}
      >
        {React.cloneElement(chosenOne, {
          cell: this.props.cell,
          id: this.props.id,
          status: this.props.status, 
          contentRef: this.props.contentRef
        })}
      </div>
    );
  }
}

export const makeMapStateToProps = (
  initialState: AppState,
  ownProps: ComponentProps
): ((state: AppState) => StateProps) => {
  const mapStateToProps = (state: AppState): StateProps => {
    const { id, contentRef } = ownProps;
    const model = selectors.model(state, { contentRef });
    let cell;
    let selected = false;
    let status = 'idle';

    if (model && model.type === "notebook") {
      cell = selectors.notebook.cellById(model, { id });
      selected = selectors.notebook.cellFocused(model) === id;
      status = model.transient.getIn(["cellMap", id, "status"]);
    }

    return { cell, selected, status };
  };
  return mapStateToProps;
};

export default connect<StateProps, void, ComponentProps, AppState>(
  makeMapStateToProps
)(Cell);
